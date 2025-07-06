import 'reflect-metadata';
import { container } from 'tsyringe';

if (process.env.NODE_ENV === 'production') {
  require('./production');
} else if (process.env.NODE_ENV === 'test') {
  require('./test');
} else {
  require('./development');
}

export { container };
