# Web frontend on Vercel. Requires VERCEL_API_TOKEN in the environment.
terraform {
  required_version = ">= 1.6"
  required_providers {
    vercel = { source = "vercel/vercel", version = ">= 2.0" }
  }
}

variable "team_id" {
  type    = string
  default = null
}
variable "github_repo" {
  type    = string
  default = "DeAtHfIrE26/Next-Generation-Virtual-Interview-Training-System"
}
variable "api_base_url" { type = string }

provider "vercel" {
  team = var.team_id
}

resource "vercel_project" "web" {
  name           = "interview-coach-web"
  framework      = "nextjs"
  root_directory = "apps/web"
  git_repository = {
    type              = "github"
    repo              = var.github_repo
    production_branch = "main"
  }
  serverless_function_region = "bom1" # Mumbai, next to the API
}

resource "vercel_project_environment_variable" "api" {
  project_id = vercel_project.web.id
  key        = "API_BASE_URL"
  value      = var.api_base_url
  target     = ["production", "preview"]
}
