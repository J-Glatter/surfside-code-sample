import 'reflect-metadata';
import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

import { handler as createJournalEntryHandler } from '../lambda/API/CreateJournalEntryHandler';
import { handler as getJournalEntriesBetweenDatesHandler } from '../lambda/API/GetJournalEntriesBetweenDatesHandler';
import { LambdaApiEvent } from '../lambda/API/LambdaApiEvent';

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());

const createApiGatewayEvent = <
  QueryStringParameters extends Record<string, unknown>,
>(
  req: express.Request,
): LambdaApiEvent<QueryStringParameters> => {
  return {
    requestContext: {
      accountId: '',
      apiId: '',
      authorizer: {
        jwt: {
          claims: {
            sub: req.query.userId as string,
          },
        },
      },
      protocol: '',
      httpMethod: '',
      identity: {
        accessKey: null,
        accountId: null,
        apiKey: null,
        apiKeyId: null,
        caller: null,
        clientCert: null,
        cognitoAuthenticationProvider: null,
        cognitoAuthenticationType: null,
        cognitoIdentityId: null,
        cognitoIdentityPoolId: null,
        principalOrgId: null,
        sourceIp: '',
        user: null,
        userAgent: null,
        userArn: null,
      },
      path: '',
      stage: '',
      requestId: '',
      requestTimeEpoch: 0,
      resourceId: '',
      resourcePath: '',
    },
    queryStringParameters: {
      ...req.query,
      ...req.params,
    } as QueryStringParameters,
    body: JSON.stringify(req.body) || '',
  };
};

// used for testing locally with the mobile app
app.use(
  cors({
    origin: 'http://localhost:8081',
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  }),
);

// note, userId will be handled in the auth, extracted from the jwt - still deciding on provider

app.post('/users/:userId/journals/:journalId/entries', async (req, res) => {
  try {
    const event = createApiGatewayEvent<{ journalId: string; userId: string }>(
      req,
    );
    const lambdaResponse = await createJournalEntryHandler(event);
    res.status(lambdaResponse.statusCode).json(JSON.parse(lambdaResponse.body));
  } catch (error: any) {
    console.error('Error in createJournalEntry handler:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/users/:userId/journals/:journalId/entries', async (req, res) => {
  try {
    const event = createApiGatewayEvent<{
      journalId: string;
      from: string;
      to: string;
      userId: string;
    }>(req);
    const lambdaResponse = await getJournalEntriesBetweenDatesHandler(event);

    res.status(lambdaResponse.statusCode).json(JSON.parse(lambdaResponse.body));
  } catch (error: any) {
    console.error('Error in GetJournalEntriesBetweenDatesHandler:', error);
    res.status(500).json({ error: error.message });
  }
});

if (process.env.NODE_ENV !== 'production') {
  app.listen(port, () => {
    console.log(`Development API server listening on port ${port}`);
  });
}

export default app;
