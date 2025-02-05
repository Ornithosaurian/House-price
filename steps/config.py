from pydantic import BaseModel
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

class ModelNameConfig(BaseModel):
    """
    Model config
    """
    name_of_model: str = "LinearRegression"

class DeploymentTriggerConfig(BaseModel):
    min_accuracy: float = 0.9

def get_docker_settings():
    docker_settings = {
    "orchestrator": LocalDockerOrchestratorSettings(
        run_args={
            "cpu_count": 5,
            "volumes": {
                "/c/Users/arkad/AppData/Roaming/zenml/local_stores": {
                    "bind": "/app/zenml_local_stores",
                    "mode": "rw",
                },
                "/f/machine learning/House Prices/data": {  
                    "bind": "/app/data",
                    "mode": "rw",
                }
            }
        }
    )
}
    return docker_settings