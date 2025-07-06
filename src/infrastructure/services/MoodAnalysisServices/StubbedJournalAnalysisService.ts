import { Analysis, Journal, JournalAnalysisService } from '../../../domain';

export class StubbedJournalAnalysisService implements JournalAnalysisService {
  constructor() {}

  async analyseEntry(
    _journal: Journal,
    _currentEntry: string,
  ): Promise<Analysis> {
    return {
      summary:
        'Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore',
    };
  }
}
