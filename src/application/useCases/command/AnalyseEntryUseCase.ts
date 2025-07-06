import {
  JournalRepository,
  JournalAnalysisService,
  JournalEntryRepository,
  Journal,
} from '../../../domain';
import { EntityNotFound } from '../../errors';
import { inject, injectable } from 'tsyringe';
import { ValidatedUseCase } from '../UseCase';
import { z } from 'zod';

type Payload = {
  userId: string;
  journalId: string;
  entryId: string;
  createdAt: string;
};

@injectable()
export class AnalyseEntryUseCase extends ValidatedUseCase<Payload, any> {
  schema = z.object({
    userId: z.string(),
    journalId: z.string(),
    entryId: z.string(),
    entryContent: z.string(),
    createdAt: z.string(),
  });

  constructor(
    @inject('JournalRepository')
    private readonly journalRepository: JournalRepository,
    @inject('JournalEntryRepository')
    private readonly journalEntryRepository: JournalEntryRepository,
    @inject('JournalAnalysisService')
    private readonly journalAnalysisService: JournalAnalysisService,
  ) {
    super();
  }

  async run(payload: Payload): Promise<void> {
    const { userId, journalId, entryId, createdAt } = payload;

    const journalWithoutEntries = await this.journalRepository.findById(
      { journalId, userId },
      false,
    );

    if (!journalWithoutEntries) {
      throw new EntityNotFound({
        entity: { name: 'Journal', id: journalId },
        message: `Journal with id ${journalId} was not found`,
      });
    }

    const journalEntry = await this.journalEntryRepository.findEntry({
      userId,
      journalId,
      entryId,
      createdAt,
    });

    if (!journalEntry)
      throw new EntityNotFound({
        entity: { name: 'JournalEntry', id: entryId },
        message: 'There is no journal entry',
        context: payload,
      });

    const temporaryJournal = Journal.fromPartial({
      ...journalWithoutEntries?.details,
      entries: [journalEntry],
    });

    const analysis = await this.journalAnalysisService.analyseEntry(
      temporaryJournal,
      journalEntry.entryContent,
    );

    temporaryJournal.analyseEntry(entryId, analysis);

    await this.journalRepository.save(temporaryJournal);
  }
}
