import 'reflect-metadata';
import { CreateJournalEntryUseCase } from '../../application/useCases/command';
import { container } from '../../di';
import { ApiHandler } from './ApiHandler';
import { LambdaApiEvent } from './LambdaApiEvent';

type CreateJournalEntryRequestParams = {
  journalId: string;
};

export class CreateJournalEntryHandler extends ApiHandler {
  public handler = async (
    event: LambdaApiEvent<CreateJournalEntryRequestParams>,
  ) => {
    const payload = {
      ...JSON.parse(event.body),
      ...event.queryStringParameters,
    };

    const createJournalEntryUseCase = container.resolve(
      CreateJournalEntryUseCase,
    );

    return this.handleRequest(() => createJournalEntryUseCase.execute(payload));
  };
}

export const handler = new CreateJournalEntryHandler().handler;
