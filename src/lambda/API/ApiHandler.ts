import { APIGatewayProxyResult } from 'aws-lambda';
import { EntityNotFound, InvalidPayload } from '../../application/errors';

export abstract class ApiHandler {
  private mapErrorToStatusCode(error: any): number {
    if (error instanceof InvalidPayload) {
      return 400;
    }

    if (error instanceof EntityNotFound) {
      return 404;
    }
    return error.statusCode || 500;
  }

  protected async handleRequest<T>(
    fn: () => Promise<T>,
  ): Promise<APIGatewayProxyResult> {
    try {
      const result = await fn();

      return {
        statusCode: 200,
        body: JSON.stringify(result),
      };
    } catch (error: any) {
      console.error('Error processing request:', error);

      return {
        statusCode: this.mapErrorToStatusCode(error),
        body: JSON.stringify({
          error,
        }),
      };
    }
  }
}
