from django.test import TestCase
from rest_framework.test import APITestCase
from mongoengine import connect, disconnect, get_connection
from django.conf import settings

class BaseMongoTestCase(APITestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        disconnect()
        connect(
            db='bakis_test_db',
            host='localhost',
            port=27017,
            username='',
            password='',
            authentication_source='admin'
        )

    @classmethod
    def tearDownClass(cls):
        from mongoengine.connection import get_connection
        conn = get_connection()
        conn.drop_database('bakis_test_db')
        disconnect()
        super().tearDownClass()
