export class Aggregate<Events> {
  protected _uncommittedEvents: Events[] = [];

  addEvent(event: Events) {
    this._uncommittedEvents.push(event);
  }

  get uncommittedEvents() {
    return this._uncommittedEvents;
  }

  clearEvents() {
    this._uncommittedEvents = [];
  }
}
