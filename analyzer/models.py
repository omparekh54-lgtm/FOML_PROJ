from django.db import models


class XrayAnalysis(models.Model):
    """One uploaded X-ray and the result of running it through the ML pipeline."""

    STATUS_CHOICES = [
        ("ok", "Analyzed"),
        ("unsupported_body_part", "Unsupported body part"),
        ("models_not_trained", "Model not trained yet"),
        ("error", "Error"),
    ]

    image = models.ImageField(upload_to="xrays/%Y/%m/%d/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    status = models.CharField(max_length=32, choices=STATUS_CHOICES, default="ok")

    body_part = models.CharField(max_length=64, blank=True, null=True)
    body_part_confidence = models.FloatField(blank=True, null=True)

    abnormality_label = models.CharField(max_length=32, blank=True, null=True)
    abnormality_confidence = models.FloatField(blank=True, null=True)

    explanation = models.TextField(blank=True, null=True)
    message = models.TextField(blank=True, null=True)  # user-facing note for non-"ok" statuses

    class Meta:
        ordering = ["-uploaded_at"]
        verbose_name_plural = "X-ray analyses"

    def __str__(self):
        return f"XrayAnalysis #{self.pk} ({self.status}, {self.uploaded_at:%Y-%m-%d %H:%M})"
