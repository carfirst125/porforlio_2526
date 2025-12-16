# =====================================================================
# Phase 2: Deploy Azure Container App from ACR image
# =====================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}

############################################################
# Variables
############################################################
variable "acr_login_server" {
  description = "ACR login server (e.g., acrname.azurecr.io)"
  type        = string
  default     = "acrngothanhnhan125.azurecr.io"
}

variable "image_repository" {
  description = "ACR repository name"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "acr_username" {
  description = "ACR admin username"
  type        = string
}

variable "acr_password" {
  description = "ACR admin password"
  type        = string
}

############################################################
# Locals
############################################################
locals {
  project_name = "agentic_chatbot"
  location     = "southeastasia"

  image_full = "${var.acr_login_server}/${var.image_repository}:${var.image_tag}"

  tags = {
    project     = local.project_name
    environment = "dev"
    managed_by  = "Terraform"
  }
}

############################################################
# Data Sources
############################################################
data "azurerm_resource_group" "rg" {
  name = local.project_name
}

data "azurerm_container_app_environment" "env" {
  name                = "aca-chatbot-env"
  resource_group_name = data.azurerm_resource_group.rg.name
}

############################################################
# Container App
############################################################
resource "azurerm_container_app" "app" {
  name                         = "agentic-chatbot-api"
  resource_group_name          = data.azurerm_resource_group.rg.name
  container_app_environment_id = data.azurerm_container_app_environment.env.id
  revision_mode                = "Single"
  tags                         = local.tags

  identity {
    type = "SystemAssigned"
  }

  registry {
    server               = var.acr_login_server
    username             = var.acr_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = var.acr_password
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    transport        = "auto"

    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    container {
      name   = "agentic-chatbot-api"
      image  = local.image_full
      cpu    = 0.5
      memory = "1Gi"
    }
  }
}

############################################################
# Outputs
############################################################
output "aca_url" {
  value = azurerm_container_app.app.latest_revision_fqdn
}

output "deployed_image" {
  value = local.image_full
}
