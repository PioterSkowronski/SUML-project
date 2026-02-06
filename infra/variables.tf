variable "location" {
  description = "The Azure region where resources will be created"
  type        = string
  default     = "polandcentral"
}

variable "resource_group_name" {
  description = "The name of the resource group"
  type        = string
  default     = "rg-weatherapp-dev-plc"
}

variable "acr_name" {
  description = "The name of the Azure Container Registry"
  type        = string
  default     = "acrweatherappdevplc"
}

variable "container_app_name" {
  description = "The name of the Container App"
  type        = string
  default     = "ca-weatherapp-dev-plc"
}

variable "container_app_environment_name" {
  description = "The name of the Container App Environment"
  type        = string
  default     = "cae-weatherapp-dev-plc"
}
