from django.conf.urls import url,include
from leaf_classify_app import views
from leaf_classify_app.views import *
from django.contrib.auth import views as auth_views
from django.conf.urls import url

urlpatterns = [
	url(r'^home/',home),
	url(r'^upload/',views.upload,name="upload"),
	url(r'home/',views.home,name="home"),
	url(r'^login/$',login,name='login'),
	url(r'^about/$',about,name='about'),
	url(r'^contact/$',contact,name='contact'),
	url(r'index2/',index2,name="index2"),
	url(r'^auth/$',auth_view),
    url('^signup/$',signup,name='signup'),
    url('^adduserinfo/$',adduserinfo),
    url('^getuserinfo/$', getuserinfo),
	url('^forgot/$', forgot),
    url('^logout/$',logout,name='logout'),
    url('^invalidlogin/$',invalidlogin),
    url('^invalidpassword/$', invalidpassword),
    url('^newpassword/$', newpassword),
	#url(r'^index/',home),
	url(r'^$',home),
]
