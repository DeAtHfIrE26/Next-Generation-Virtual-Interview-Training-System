variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "asia-south1"
}
variable "api_image" {
  type        = string
  description = "e.g. asia-south1-docker.pkg.dev/<project>/interview-coach/api:<git-sha>"
}
variable "app_base_url" { type = string }
variable "db_tier" {
  type    = string
  default = "db-custom-1-3840"
}
variable "db_high_availability" {
  type    = bool
  default = false
}
variable "api_min_instances" {
  type    = number
  default = 0
}
variable "api_max_instances" {
  type    = number
  default = 10
}
variable "biometric_retention_days" {
  type    = number
  default = 30
}
variable "llm_provider" {
  type    = string
  default = "anthropic"
}
variable "llm_model" {
  type    = string
  default = "claude-opus-5-5"
}
variable "asr_provider" {
  type    = string
  default = "browser"
}
variable "tts_provider" {
  type    = string
  default = "browser"
}
variable "otel_endpoint" {
  type    = string
  default = ""
}
