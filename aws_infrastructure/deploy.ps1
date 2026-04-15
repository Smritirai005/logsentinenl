$PROJECT_NAME = "log-anomaly-detection"
$STACK_NAME = "$PROJECT_NAME-stack"
$REGION = "us-east-1"
$ALERT_EMAIL = "your-email@example.com"

Write-Host "Deploying $STACK_NAME to $REGION"

aws cloudformation deploy `
  --template-file cloudformation.yaml `
  --stack-name $STACK_NAME `
  --parameter-overrides `
      ProjectName=$PROJECT_NAME `
      AlertEmail=$ALERT_EMAIL `
  --capabilities CAPABILITY_NAMED_IAM `
  --region $REGION

Write-Host "Stack deployed successfully!"

Write-Host "`nStack Outputs:"
aws cloudformation describe-stacks `
  --stack-name $STACK_NAME `
  --region $REGION `
  --query "Stacks[0].Outputs" `
  --output table

Write-Host "`nDeployment complete!"