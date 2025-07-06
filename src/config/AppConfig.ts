import convict from 'convict';

const configSchema = {
  tableName: {
    doc: 'DynamoDB table name',
    format: String,
    default: '',
    env: 'TABLE_NAME',
  },
  region: {
    doc: 'AWS Region',
    format: String,
    default: 'eu-west-1',
    env: 'AWS_REGION',
  },

  dynamoDBEndpoint: {
    doc: 'DynamoDB endpoint',
    format: String,
    default: '',
    env: 'DYNAMODB_ENDPOINT',
  },
};

const config = convict(configSchema);

config.validate({ allowed: 'strict' });

export type AppConfig = Record<keyof typeof configSchema, unknown>;

const appConfig: AppConfig = config.getProperties();
export default appConfig;
