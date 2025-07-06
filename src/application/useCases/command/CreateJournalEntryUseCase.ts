import {
  JournalEntry,
  JournalEntryProps,
  JournalRepository,
} from '../../../domain';
import { randomUUID } from 'node:crypto';
import { EntityNotFound } from '../../errors';
import { inject, injectable } from 'tsyringe';
import { ValidatedUseCase } from '../UseCase';
import { z } from 'zod';

type Payload = {
  userId: string;
  journalId: string;
  entryContent: string;
  tags?: string[];
};

type Response = { newEntry: JournalEntryProps };

@injectable()
export class CreateJournalEntryUseCase extends ValidatedUseCase<
  Payload,
  Response
> {
  schema = z.object({
    userId: z.string(),
    journalId: z.string(),
    entryContent: z.string(),
    tags: z.array(z.string()).optional(),
  });

  constructor(
    @inject('JournalRepository')
    private readonly journalRepository: JournalRepository,
  ) {
    super();
  }

  async run(payload: Payload): Promise<Response> {
    const { userId, journalId, entryContent, tags = [] } = payload;

    const journal = await this.journalRepository.findById(
      { journalId, userId },
      false,
    );

    if (!journal) {
      throw new EntityNotFound({
        entity: { name: 'Journal', id: journalId },
        message: `Journal with id ${journalId} was not found`,
      });
    }

    const newEntry = JournalEntry.create({
      entryId: randomUUID(),
      userId,
      journalId,
      entryContent,
      tags,
    });

    journal.addEntry(newEntry);
    await this.journalRepository.save(journal);

    return { newEntry: newEntry.details };
  }
}
