try:
    from mlflow import log_metric
except ImportError:
    def log_metric(name, value, step):
        print(f"Step {step}: {name} == {value}")


def log_(name, value, step):
    log_metric(name, value, step)
