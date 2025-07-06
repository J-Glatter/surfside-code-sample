import { JournalEntry, JournalEntryProps } from '../../../domain';
import { ValidatedUseCase } from '../UseCase';
import { inject, injectable } from 'tsyringe';
import { z } from 'zod';
import { JournalReadRepository } from '../../queries';

type Payload = {
  userId: string;
  journalId: string;
  from: string;
  to: string;
};

type Response = {
  entries: JournalEntryProps[];
};

@injectable()
export class GetJournalEntriesBetweenDatesUseCase extends ValidatedUseCase<
  Payload,
  Response
> {
  public schema = z.object({
    userId: z.string(),
    journalId: z.string(),
    from: z
      .string()
      .regex(/^\d{4}-\d{2}-\d{2}$/, 'Invalid date, expected YYYY-MM-DD'),
    to: z
      .string()
      .regex(/^\d{4}-\d{2}-\d{2}$/, 'Invalid date, expected YYYY-MM-DD'),
  });

  constructor(
    @inject('JournalReadRepository')
    private readonly journalRepository: JournalReadRepository,
  ) {
    super();
  }

  public async run({
    userId,
    journalId,
    from,
    to,
  }: Payload): Promise<Response> {
    const formattedFrom = from.includes('T') ? from.split('T')[0] : from;
    const formattedTo = to.includes('T') ? to.split('T')[0] : to;

    const fromDate = new Date(`${formattedFrom}T00:00:00Z`);
    const toDate = new Date(`${formattedTo}T23:59:59Z`);

    const entries = await this.journalRepository.findEntries(
      journalId,
      userId,
      {
        between: { fromDate, toDate },
      },
    );

    return { entries: entries.map((entry) => entry.details) };
  }
}
