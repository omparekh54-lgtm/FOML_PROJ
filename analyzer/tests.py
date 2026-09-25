from django.test import TestCase

class AnalyzerSmokeTest(TestCase):
    def test_homepage(self):
        response = self.client.get("/")
        self.assertIn(response.status_code, (200, 302))
