import { randomUUID } from 'node:crypto';

export class DomainEvent<Payload> {
  public readonly id: string;
  public readonly payload: Payload;
  public readonly eventName: string;
  public readonly occurredAt: Date;

  public constructor(payload: Payload, eventName: string) {
    this.eventName = eventName;
    this.payload = payload;
    this.occurredAt = new Date();
    this.id = randomUUID();
  }
}
