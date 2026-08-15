# Infrastructure as Code (Terraform)

Provisions the AWS resources for two of the app's compute patterns:

1. **ECS Fargate** - runs the FastAPI app (this repo's own `Dockerfile`) as an
   always-on service inside a dedicated VPC with two public subnets.
2. **Lambda + API Gateway (HTTP API)** - a second, HTTP-triggerable ingestion
   entrypoint (`src/infra/aws/api_trigger_handler.py`), alongside the
   S3-event-triggered Lambda already deployed via AWS SAM (`../../template.yaml`).

This is a **demonstration IaC module**, not a drop-in production deploy - see
the caveats in `main.tf`'s comments, in particular:

- The Lambda deployment package only zips the handler module itself; it does
  **not** bundle `boto3`'s transitive dependencies or the `llama-index`/RAG
  stack from `../../requirements.txt`. A real deploy needs either a Lambda
  Layer for those dependencies or `package_type = "Image"` (container-image
  Lambda). The S3-triggered ingestion path already handles this correctly via
  `sam build`.
- No ALB in front of the ECS service - the task gets a public IP directly for
  simplicity. Add an Application Load Balancer + target group before using
  this for anything beyond a demo/interview walkthrough.
- Never run against a real AWS account with `terraform apply` blindly - review
  the plan output first (`terraform plan`), same as any infrastructure change.

## Usage

```bash
terraform init
terraform validate
terraform plan -var="container_image=<your-ecr-repo>:<tag>"
```

`container_image` is the only required variable with no default - build and
push the image first:

```bash
docker build -t <your-ecr-repo>:<tag> ..
docker push <your-ecr-repo>:<tag>
```
