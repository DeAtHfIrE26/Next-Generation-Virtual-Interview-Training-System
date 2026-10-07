# API, database, keys and scheduled jobs on Google Cloud (region asia-south1, Mumbai).
# Secrets are created empty; set their values out of band (gcloud secrets versions add ...).

terraform {
  required_version = ">= 1.6"
  required_providers {
    google = { source = "hashicorp/google", version = ">= 6.0" }
  }
  backend "gcs" {} # configure with -backend-config="bucket=..." at init
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  secrets = ["database-url", "llm-api-key", "deepgram-api-key", "stripe-secret-key", "stripe-webhook-secret",
  "razorpay-key-secret", "razorpay-webhook-secret", "metrics-token"]
}

resource "google_project_service" "apis" {
  for_each = toset(["run.googleapis.com", "sqladmin.googleapis.com", "secretmanager.googleapis.com",
  "cloudkms.googleapis.com", "artifactregistry.googleapis.com", "cloudscheduler.googleapis.com"])
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "interview-coach"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

resource "google_service_account" "api" {
  account_id   = "interview-api"
  display_name = "Interview Coach API"
}

# --- Database ---------------------------------------------------------------------------
resource "google_sql_database_instance" "pg" {
  name             = "interview-pg"
  database_version = "POSTGRES_16"
  region           = var.region
  settings {
    tier              = var.db_tier
    availability_type = var.db_high_availability ? "REGIONAL" : "ZONAL"
    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = true
    }
    ip_configuration {
      ipv4_enabled = true
      ssl_mode     = "ENCRYPTED_ONLY"
    }
    database_flags {
      name  = "log_min_duration_statement"
      value = "500"
    }
  }
  deletion_protection = true
  depends_on          = [google_project_service.apis]
}

resource "google_sql_database" "app" {
  name     = "interview"
  instance = google_sql_database_instance.pg.name
}

# --- Keys and secrets --------------------------------------------------------------------
resource "google_kms_key_ring" "ring" {
  name       = "interview-coach"
  location   = var.region
  depends_on = [google_project_service.apis]
}

resource "google_kms_crypto_key" "templates" {
  name            = "biometric-templates"
  key_ring        = google_kms_key_ring.ring.id
  rotation_period = "7776000s" # 90 days; old versions stay enabled to unwrap existing data keys
  lifecycle { prevent_destroy = true }
}

resource "google_kms_crypto_key_iam_member" "api_templates" {
  crypto_key_id = google_kms_crypto_key.templates.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:${google_service_account.api.email}"
}

resource "google_secret_manager_secret" "s" {
  for_each  = toset(local.secrets)
  secret_id = "interview-${each.value}"
  replication {
    user_managed {
      replicas { location = var.region }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "api" {
  for_each  = google_secret_manager_secret.s
  secret_id = each.value.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "api_sql" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}

# --- API on Cloud Run (scales to zero) -----------------------------------------------------
resource "google_cloud_run_v2_service" "api" {
  name     = "interview-api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"
  template {
    service_account = google_service_account.api.email
    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = var.api_max_instances
    }
    max_instance_request_concurrency = 40
    timeout                          = "60s"
    volumes {
      name = "cloudsql"
      cloud_sql_instance { instances = [google_sql_database_instance.pg.connection_name] }
    }
    containers {
      image = var.api_image
      resources {
        limits = { cpu = "1", memory = "1Gi" }
      }
      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
      dynamic "env" {
        for_each = {
          APP_ENV                     = "production"
          APP_BASE_URL                = var.app_base_url
          TEMPLATE_KMS_KEY            = google_kms_crypto_key.templates.id
          BIOMETRIC_RETENTION_DAYS    = tostring(var.biometric_retention_days)
          LLM_PROVIDER                = var.llm_provider
          LLM_MODEL                   = var.llm_model
          ASR_PROVIDER                = var.asr_provider
          TTS_PROVIDER                = var.tts_provider
          FEATURE_B2B_HIRING          = "false"
          OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_endpoint
        }
        content {
          name  = env.key
          value = env.value
        }
      }
      dynamic "env" {
        for_each = {
          DATABASE_URL            = "database-url"
          ANTHROPIC_API_KEY       = "llm-api-key"
          DEEPGRAM_API_KEY        = "deepgram-api-key"
          STRIPE_SECRET_KEY       = "stripe-secret-key"
          STRIPE_WEBHOOK_SECRET   = "stripe-webhook-secret"
          RAZORPAY_KEY_SECRET     = "razorpay-key-secret"
          RAZORPAY_WEBHOOK_SECRET = "razorpay-webhook-secret"
          METRICS_TOKEN           = "metrics-token"
        }
        content {
          name = env.key
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.s[env.value].secret_id
              version = "latest"
            }
          }
        }
      }
    }
  }
  depends_on = [google_secret_manager_secret_iam_member.api, google_project_iam_member.api_sql]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers" # the API enforces its own auth; the web app proxies to it
}

# --- Daily retention purge ------------------------------------------------------------------
resource "google_cloud_run_v2_job" "purge" {
  name     = "interview-retention-purge"
  location = var.region
  template {
    template {
      service_account = google_service_account.api.email
      volumes {
        name = "cloudsql"
        cloud_sql_instance { instances = [google_sql_database_instance.pg.connection_name] }
      }
      containers {
        image   = var.api_image
        command = ["python", "-m", "interview_api.jobs", "purge"]
        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }
        env {
          name = "DATABASE_URL"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.s["database-url"].secret_id
              version = "latest"
            }
          }
        }
        env {
          name  = "APP_ENV"
          value = "production"
        }
      }
    }
  }
}

resource "google_service_account" "scheduler" {
  account_id = "interview-scheduler"
}

resource "google_cloud_run_v2_job_iam_member" "scheduler" {
  name     = google_cloud_run_v2_job.purge.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

resource "google_cloud_scheduler_job" "purge" {
  name      = "interview-retention-purge-daily"
  region    = var.region
  schedule  = "17 3 * * *"
  time_zone = "Asia/Kolkata"
  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${var.project_id}/jobs/${google_cloud_run_v2_job.purge.name}:run"
    oauth_token { service_account_email = google_service_account.scheduler.email }
  }
}

output "api_url" { value = google_cloud_run_v2_service.api.uri }
output "templates_kms_key" { value = google_kms_crypto_key.templates.id }
