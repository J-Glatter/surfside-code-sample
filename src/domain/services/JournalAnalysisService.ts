import { Journal } from '../aggregates';
import { Analysis } from '../valueObjects';

export interface JournalAnalysisService {
  analyseEntry(journal: Journal, currentEntry: string): Promise<Analysis>;
}
