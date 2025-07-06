import { APIGatewayEventRequestContext } from 'aws-lambda';

export type LambdaApiEvent<
  QueryStringParameters extends Record<string, unknown>,
> = {
  body: string;
  queryStringParameters: QueryStringParameters;
  requestContext: APIGatewayEventRequestContext;
};
