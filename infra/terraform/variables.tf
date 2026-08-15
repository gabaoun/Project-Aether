variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix applied to all resource names/tags."
  type        = string
  default     = "project-aether"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for the two public subnets (across two AZs, required by the ALB/ECS service)."
  type        = list(string)
  default     = ["10.20.1.0/24", "10.20.2.0/24"]
}

variable "container_image" {
  description = "Fully-qualified image URI for the API container (built from the repo's own Dockerfile, pushed to ECR)."
  type        = string
}

variable "container_port" {
  description = "Port the FastAPI app listens on inside the container (see Dockerfile / uvicorn command)."
  type        = number
  default     = 8000
}

variable "fargate_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)."
  type        = number
  default     = 256
}

variable "fargate_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Number of running ECS tasks."
  type        = number
  default     = 1
}
