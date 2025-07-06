import 'reflect-metadata';
import { container } from '../../di';
import { GetJournalEntriesBetweenDatesUseCase } from '../../application/useCases/query';
import { ApiHandler } from './ApiHandler';
import { APIGatewayProxyResult } from 'aws-lambda';
import { LambdaApiEvent } from './LambdaApiEvent';

type GetJournalEntriesBetweenDatesRequestParams = {
  journalId: string;
  from: string;
  to: string;
  userId: string;
};

export class GetJournalEntriesBetweenDatesHandler extends ApiHandler {
  public handler = async (
    event: LambdaApiEvent<GetJournalEntriesBetweenDatesRequestParams>,
  ): Promise<APIGatewayProxyResult> => {
    const payload = { ...event.queryStringParameters };

    const getJournalEntriesBetweenDatesUseCase = container.resolve(
      GetJournalEntriesBetweenDatesUseCase,
    );

    return this.handleRequest(() =>
      getJournalEntriesBetweenDatesUseCase.execute(payload),
    );
  };
}

export const handler = new GetJournalEntriesBetweenDatesHandler().handler;
