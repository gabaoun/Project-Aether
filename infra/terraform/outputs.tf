output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "ingestion_trigger_api_endpoint" {
  description = "POST here with {\"bucket\": ..., \"key\": ...} to trigger ingestion on demand."
  value       = "${aws_apigatewayv2_api.ingestion.api_endpoint}/ingest-trigger"
}
