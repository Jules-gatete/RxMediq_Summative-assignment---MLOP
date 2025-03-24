from locust import HttpUser, task, between

class PredictionUser(HttpUser):
    wait_time = between(1, 5)
    
    @task
    def predict(self):
        self.client.post("/predict", json={
            "disease": "diarrhea",
            "age": 30,
            "gender": "male",
            "severity": "NORMAL"
        })