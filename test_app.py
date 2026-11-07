import unittest
from app import app
import database as db

class FactGuardIntegrationTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_01_landing_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SemanticTruth", response.data)
        self.assertIn(b"Vipul Yadav", response.data)
        self.assertIn(b"CUSAT", response.data)
        print("[PASS] Landing page loads correctly.")

    def test_02_detector_page_get(self):
        response = self.client.get("/detector")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Instant News & Link Credibility Checker", response.data)
        self.assertNotIn(b"model-selector", response.data)
        print("[PASS] Detector page GET loads without 3-model option dropdown.")

    def test_03_realtime_api_real_and_fake(self):
        # 1. Real News Test (General science / medical reporting)
        real_text = "The World Health Organization confirmed on Tuesday that seasonal influenza vaccinations reduced clinical hospitalizations by 43 percent across participating European healthcare facilities, according to a peer-reviewed study published this week. Officials stated that ongoing surveillance efforts will monitor antibody persistence throughout the winter season."
        res_real = self.client.post("/api/analyze", json={
            "news": real_text,
            "url": ""
        })
        self.assertEqual(res_real.status_code, 200)
        data_real = res_real.get_json()
        self.assertTrue(data_real["success"])
        self.assertTrue(data_real["result"]["is_real"])
        self.assertEqual(data_real["result"]["status"], "real")
        self.assertIn("confirmed", data_real["result"]["basis_explanation"].lower())
        print(f"[PASS] Real news correctly classified as REAL ({data_real['result']['confidence']}% confidence)")
        print(f"       Basis: {data_real['result']['basis_explanation']}")

        # 2. Fake News Test (Sensational conspiracy clickbait)
        fake_text = "SHOCKING SECRET EXPOSED: Top scientists discover secret chemical lasers installed in underground government bunkers to control human minds and manipulate elections! The mainstream media elites are desperately trying to censor this whistleblower alert! SHARE THIS IMMEDIATELY BEFORE IT IS BANNED FOREVER!!!"
        res_fake = self.client.post("/api/analyze", json={
            "news": fake_text,
            "url": ""
        })
        self.assertEqual(res_fake.status_code, 200)
        data_fake = res_fake.get_json()
        self.assertTrue(data_fake["success"])
        self.assertFalse(data_fake["result"]["is_real"])
        self.assertEqual(data_fake["result"]["status"], "fake")
        self.assertIn("clickbait", data_fake["result"]["basis_explanation"].lower())
        print(f"[PASS] Fake news correctly classified as FAKE ({data_fake['result']['confidence']}% confidence)")
        print(f"       Basis: {data_fake['result']['basis_explanation']}")

    def test_04_user_registration_and_logged_in_audit(self):
        username = "realtimeuser"
        email = "rtuser@factguard.org"
        password = "FactGuardRealTime2026!"

        conn = db.get_db()
        conn.execute("DELETE FROM users WHERE username = ? OR email = ?", (username, email))
        conn.commit()
        conn.close()

        # Register
        res_reg = self.client.post("/register", data={
            "username": username,
            "email": email,
            "password": password,
            "confirm_password": password
        }, follow_redirects=True)
        self.assertEqual(res_reg.status_code, 200)

        # Login
        res_login = self.client.post("/login", data={
            "identifier": username,
            "password": password
        }, follow_redirects=True)
        self.assertEqual(res_login.status_code, 200)

        # Analyze as logged in user
        scan_text = "British health authorities confirmed regulatory clearance for a newly approved preventative medication today following successful trials, according to a department spokesperson."
        res_scan = self.client.post("/detector", data={
            "news": scan_text,
            "url": ""
        })
        self.assertEqual(res_scan.status_code, 200)

        # Check dashboard
        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn(b"realtimeuser", res_dash.data)
        self.assertIn(b"Total Articles Analyzed", res_dash.data)
        print("[PASS] User registration, login, and logged-in real-time scan verified.")

    def test_05_health_api(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        print("[PASS] Health API returns healthy.")

if __name__ == "__main__":
    unittest.main()
