import { ApplicationError } from './ApplicationError';

type ErrorProps = {
  message: string;
  errors?: string[];
};

export class InvalidPayload extends ApplicationError {
  name = 'InvalidPayload';

  constructor({ errors, message }: ErrorProps) {
    super(InvalidPayload.name, `Invalid Payload ${message}`, { errors });
  }
}
