from django.contrib import admin
from .models import XrayAnalysis

@admin.register(XrayAnalysis)
class XrayAnalysisAdmin(admin.ModelAdmin):
    list_display=("id","uploaded_at","status","body_part","body_part_confidence","abnormality_label","abnormality_confidence")
    list_filter=("status","body_part","abnormality_label")
    readonly_fields=("uploaded_at","status","body_part","body_part_confidence","abnormality_label","abnormality_confidence","explanation","message")
