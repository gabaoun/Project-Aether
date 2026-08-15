# Project Aether - infrastructure as code
#
# Two things this module provisions, matching the two AWS compute patterns
# used by the app: an always-on API (ECS Fargate) and an event-driven
# ingestion pipeline (Lambda). The Lambda's real trigger in production is
# the S3 upload event already declared in ../../template.yaml (AWS SAM) -
# this module additionally fronts a *second*, HTTP-triggerable Lambda with
# API Gateway to demonstrate that IaC pattern explicitly (manual/ad-hoc
# ingestion trigger, distinct from the S3-driven one).
#
# Not yet applied against a real AWS account - written and reviewed by hand
# (no `terraform` CLI in the authoring environment to run `init`/`validate`).
# Run `terraform init && terraform validate` before the first `apply`.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_availability_zones" "available" {
  state = "available"
}

# ---------------------------------------------------------------------------
# Networking: VPC + public subnets
# ---------------------------------------------------------------------------

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-${count.index}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------------------
# ECS Fargate: hosts the FastAPI app (src/api/app.py, built via the repo's
# Dockerfile) as an always-on service, replacing the free-tier Render deploy
# for a production-shaped setup with its own VPC-scoped networking.
# ---------------------------------------------------------------------------

resource "aws_security_group" "ecs_service" {
  name_prefix = "${var.project_name}-ecs-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "API traffic"
    from_port   = var.container_port
    to_port     = var.container_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-ecs-sg"
  }
}

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14
}

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${var.project_name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.fargate_cpu
  memory                   = var.fargate_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${var.project_name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs_service.id]
    assign_public_ip = true
  }
}

# ---------------------------------------------------------------------------
# Lambda + API Gateway: HTTP-triggerable ingestion, alongside the existing
# S3-triggered path in template.yaml.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Packaged separately from the SAM-managed ingestion Lambda (src/infra/aws/
# lambda_handler.py expects an S3 event shape). This one's handler:
# src/infra/aws/api_trigger_handler.py.
#
# NOTE: this only zips the handler module itself, matching the "basic IaC
# demonstration" scope this was written for - it does NOT bundle boto3's
# transitive deps or llama-index (../../requirements.txt), which is why
# the S3-triggered Lambda in ../../template.yaml is deployed via
# `sam build` instead (it resolves and packages the full dependency tree
# from requirements.txt into the build artifact). A real `terraform apply`
# of this resource needs either a Lambda Layer for those dependencies or
# switching to a container-image Lambda (package_type = "Image").
data "archive_file" "ingestion_trigger" {
  type        = "zip"
  source_file = "${path.module}/../../src/infra/aws/api_trigger_handler.py"
  output_path = "${path.module}/build/api_trigger_handler.zip"
}

resource "aws_lambda_function" "ingestion_trigger" {
  function_name    = "${var.project_name}-ingestion-trigger"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "api_trigger_handler.handler"
  runtime          = "python3.11"
  filename         = data.archive_file.ingestion_trigger.output_path
  source_code_hash = data.archive_file.ingestion_trigger.output_base64sha256
  timeout          = 30
  memory_size      = 256
}

resource "aws_apigatewayv2_api" "ingestion" {
  name          = "${var.project_name}-ingestion-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "ingestion" {
  api_id                 = aws_apigatewayv2_api.ingestion.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.ingestion_trigger.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "ingestion" {
  api_id    = aws_apigatewayv2_api.ingestion.id
  route_key = "POST /ingest-trigger"
  target    = "integrations/${aws_apigatewayv2_integration.ingestion.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.ingestion.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion_trigger.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.ingestion.execution_arn}/*/*"
}
