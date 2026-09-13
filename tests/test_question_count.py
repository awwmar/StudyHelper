"""Exercise real Flask routes with external AI/database services replaced."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch


class QuestionCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dependencies = [
            'services.client', 'agents.system1_ingestion', 'services.vector',
            'services.llm', 'services.selector', 'agents.system2_query',
            'agents.feedback_agent', 'services.db',
        ]
        cls.dependencies = {name: MagicMock() for name in dependencies}
        with patch.dict('sys.modules', cls.dependencies):
            spec = importlib.util.spec_from_file_location(
                'question_count_app', Path(__file__).resolve().parents[1] / 'app.py')
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)
        cls.module.app.config.update(TESTING=True, SECRET_KEY='test-only')

    def setUp(self):
        self.client = self.module.app.test_client()
        with self.client.session_transaction() as session:
            session['user_id'] = 'test-user'
        self.generate = self.module.system2.generate_quiz
        self.generate.reset_mock()
        # Reproduce the exact slice that failed when passed the browser's "5".
        self.generate.side_effect = lambda **kw: [
            {'question': 'What is a stack?', 'correct': 'LIFO'}
        ][:kw['num_questions']]

    def test_browser_string_and_json_integer_work_on_both_routes(self):
        for route in ['/generate-questions', '/generate-test']:
            for count in ['5', 5]:
                with self.subTest(route=route, count=count):
                    response = self.client.post(route, json={'num_questions': count, 'save': False})
                    self.assertEqual(response.status_code, 200, response.json)
                    self.assertTrue(response.json['questions'])
                    self.assertIs(type(self.generate.call_args.kwargs['num_questions']), int)

    def test_invalid_counts_rejected_before_ai_call(self):
        for route in ['/generate-questions', '/generate-test']:
            for count in ['', 'abc', '2.5', 2.5, 0, -1, None, True, [], {}]:
                with self.subTest(route=route, count=count):
                    self.generate.reset_mock()
                    response = self.client.post(route, json={'num_questions': count, 'save': False})
                    self.assertEqual(response.status_code, 400, response.json)
                    self.assertIn('positive whole number', response.json['error'])
                    self.generate.assert_not_called()

    def test_missing_count_defaults_to_five(self):
        for route in ['/generate-questions', '/generate-test']:
            response = self.client.post(route, json={'save': False})
            self.assertEqual(response.status_code, 200, response.json)
            self.assertEqual(self.generate.call_args.kwargs['num_questions'], 5)


if __name__ == '__main__':
    unittest.main()
