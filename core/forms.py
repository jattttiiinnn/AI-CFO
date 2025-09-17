from django import forms
from django.core.exceptions import ValidationError

ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf'}


def validate_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError('Unsupported file type. Please upload a CSV, Excel, or PDF file (.csv, .xlsx, .xls, .pdf).')


class UploadForm(forms.Form):
    file = forms.FileField(
        label='Upload Financial Report (CSV/XLSX/XLS/PDF)',
        validators=[validate_file_extension],
        widget=forms.ClearableFileInput(attrs={'accept': '.csv,.xlsx,.xls,.pdf'})
    )
