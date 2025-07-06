import { mock } from 'jest-mock-extended';
import { Journal, JournalEntry, JournalRepository } from '../../../../domain';
import { CreateJournalEntryUseCase } from '../../command';
import { EntityNotFound } from '../../../errors';

const journal = mock<Journal>();
const journalEntry = JournalEntry.create({
  entryContent: 'Test Entry',
  entryId: 'cd6a3b3b-01dd-4f41-adf1-a9471e818cfc',
  journalId: '68df0c8f-fc34-4ca4-becf-4c5c8c363bc0',
  tags: [],
  userId: 'bcb328c3-8383-4fcf-a784-69852713251d',
});

const journalRepository = mock<JournalRepository>({
  findById: jest.fn().mockResolvedValue(journal),
});

const payload = {
  userId: '98f76d1e-f336-4e21-a034-6ed2cde1c537',
  journalId: '3bce0c11-62be-4856-8bbc-042d841e7ba7',
  entryContent: 'Test entry',
};

describe('CreateJournalEntryUseCase', () => {
  let useCase: CreateJournalEntryUseCase;

  beforeEach(() => {
    jest.clearAllMocks();

    useCase = new CreateJournalEntryUseCase(journalRepository);
    jest.spyOn(JournalEntry, 'create').mockReturnValue(journalEntry);
  });

  it('should get the Journal without any entries', async () => {
    await useCase.execute(payload);
    expect(journalRepository.findById).toHaveBeenCalledWith(
      { journalId: payload.journalId, userId: payload.userId },
      false,
    );
  });

  it('should throw an error if the Journal does not exist', async () => {
    journalRepository.findById.mockResolvedValueOnce(null);
    await expect(async () => await useCase.execute(payload)).rejects.toThrow(
      EntityNotFound,
    );
  });

  it('should add a new Journal Entry and save the Journal', async () => {
    await useCase.execute(payload);

    expect(journal.addEntry).toHaveBeenCalledWith(journalEntry);
    expect(journalRepository.save).toHaveBeenCalledWith(journal);
  });

  it('should return the new entry', async () => {
    const response = await useCase.execute(payload);

    expect(response).toEqual({ newEntry: journalEntry.details });
  });
});
