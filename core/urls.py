from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    # Generic Upload & Export
    path('', views.upload_view, name='upload'),
    path('export/', views.export_json_view, name='export_json'),
    path('export/pdf/', views.cfo_export_pdf_view, name='export_pdf'),  # Add this line

    # Zomato Analyzer
    path('zomato/', views.zomato_upload_view, name='zomato_upload'),
    path('zomato/export/', views.zomato_export_json_view, name='zomato_export_json'),
    path('zomato/export/pdf/', views.zomato_export_pdf_view, name='zomato_export_pdf'),

    # AI CFO Assistant
    path('cfo/analyze/', views.analyze_financials, name='cfo_analyze'),
    path('cfo/export/pdf/', views.cfo_export_pdf_view, name='cfo_export_pdf'),
]