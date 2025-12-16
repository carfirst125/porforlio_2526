# =====================================================================
# Phase 1: Terraform Infrastructure for Azure Container App (pre-image)
# Creates RG, ACR, and Container App Environment
# =====================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  backend "azurerm" {
    resource_group_name  = "terraform-backend-rg"
    storage_account_name = "tfbackendstorage125"
    container_name       = "tfstate"
    key                  = "agentic-chatbot-phase1.terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

locals {
  project_name = "agentic_chatbot"
  location     = "southeastasia"

  tags = {
    project     = local.project_name
    environment = "dev"
    owner       = "Ngo Thanh Nhan"
    managed_by  = "Terraform"
    created_at  = "2025-10-29"
  }
}

############################################################
# Resource Group for App
############################################################
resource "azurerm_resource_group" "rg" {
  name     = local.project_name
  location = local.location
  tags     = local.tags
}

############################################################
# Azure Container Registry (ACR)
############################################################
resource "azurerm_container_registry" "acr" {
  name                = "acrngothanhnhan125"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

############################################################
# Azure Container App Environment
############################################################
resource "azurerm_container_app_environment" "env" {
  name                = "aca-chatbot-env"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = local.tags
}

############################################################
# Outputs
############################################################
output "acr_login_server" {
  value = azurerm_container_registry.acr.login_server
}

output "aca_env_id" {
  value = azurerm_container_app_environment.env.id
}
