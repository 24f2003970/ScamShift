"""MLflow flavor and optional mirror of ScamShift's committed deployment."""
import hashlib
import os
from tempfile import TemporaryDirectory
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path

NAME = os.environ.get("SCAMSHIFT_REGISTERED_MODEL", "ScamShiftClassifier")
ALIAS = "champion"
EXAMPLE = ["The library closes at six today.", "Urgent: share your OTP now."]


class RegistryError(RuntimeError):
    pass


def enabled():
    return os.environ.get("SCAMSHIFT_REGISTRY") == "1"


def client_for(tracking_uri):
    registry_uri = os.environ.get("MLFLOW_REGISTRY_URI") or tracking_uri
    if enabled() and (not tracking_uri.startswith(("http://", "https://"))
                      or not registry_uri.startswith(("http://", "https://"))):
        raise RegistryError("Registry mode requires an HTTP MLflow server.")
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)


def log_flavor(client, run_id, model, report):
    import numpy as np
    from mlflow import sklearn as flavor
    from mlflow.models import infer_signature
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    from .model import normalize, robust_normalize

    normalizer = robust_normalize if report.get("preprocessing") == "robust" else normalize
    pipeline = Pipeline([
        ("scamshift_normalize", FunctionTransformer(_normalize_messages, kw_args={"normalize": normalizer})),
        *model.steps,
    ])
    samples = np.asarray(EXAMPLE)
    requirements = [
        f"{name}=={version}" for name, version in report.get("environment", {}).items()
        if name in {"scikit-learn", "numpy", "scipy", "mlflow-skinny"}
    ]
    if requirements:
        from importlib.metadata import version
        requirements.append(f"pandas=={version('pandas')}")
    with TemporaryDirectory() as folder:
        path = Path(folder) / "model"
        flavor.save_model(
            pipeline, str(path), serialization_format="pickle", pyfunc_predict_fn="predict_proba",
            signature=infer_signature(samples, pipeline.predict_proba(samples)),
            input_example=samples, pip_requirements=requirements or None,
            code_paths=[str(Path(__file__).resolve().parent)],
            metadata={"threshold": report["threshold"], "positive_class": 1,
                      "artifact_sha256": report["artifact_sha256"]},
        )
        client.log_artifacts(run_id, str(path), artifact_path="model")


def _normalize_messages(messages, normalize):
    if hasattr(messages, "reshape"):
        messages = messages.reshape(-1)
    return [normalize(message) for message in messages]


def find_version(client, identity):
    try:
        matches = client.search_model_versions(f"name = '{NAME}'")
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None
    return next((version for version in matches if version.tags.get("scamshift.identity") == identity), None)


def register(client, run_id, identity):
    try:
        client.create_registered_model(NAME)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    existing = find_version(client, identity)
    if existing is not None:
        return existing
    source = append_to_uri_path(client.get_run(run_id).info.artifact_uri, "model")
    return client.create_model_version(
        NAME, source, run_id=run_id, tags={"scamshift.identity": identity}, await_creation_for=30
    )


def identity(state_id, version, digest):
    return hashlib.sha256(f"{state_id}:{version}:{digest}".encode()).hexdigest()


def verify(client, mapping, model, metadata):
    import numpy as np
    from mlflow import sklearn as flavor
    from mlflow.models import Model

    from .model import normalize, probabilities, robust_normalize

    version = client.get_model_version(NAME, mapping["registry_version"])
    if version.run_id != mapping["registry_run"]:
        raise RegistryError("Registered MLflow version points to a different run.")
    preprocess = robust_normalize if metadata.get("preprocessing") == "robust" else normalize
    with TemporaryDirectory() as folder:
        path = client.download_artifacts(mapping["registry_run"], "model", folder)
        logged = Model.load(str(Path(path) / "MLmodel")).metadata
        if logged["artifact_sha256"] != metadata["artifact_sha256"] or logged["threshold"] != metadata["threshold"]:
            raise RegistryError("MLflow model metadata does not match the local artifact.")
        loaded = flavor.load_model(path)
        np.testing.assert_allclose(loaded.predict_proba(EXAMPLE)[:, 1], probabilities(model, EXAMPLE, preprocess=preprocess),
                                   rtol=0, atol=1e-12)


def set_alias(client, version):
    client.set_registered_model_alias(NAME, ALIAS, version)
    if str(client.get_model_version_by_alias(NAME, ALIAS).version) != str(version):
        raise RegistryError("MLflow did not confirm the champion alias.")
