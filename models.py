from django.db import models
class User(models.Model):
	uid = models.CharField(max_length=15,primary_key=True)
	uname = models.CharField(max_length=15)
	password = models.CharField(max_length=15)
	email= models.CharField(max_length=15)

	