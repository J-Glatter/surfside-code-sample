import 'reflect-metadata';
import bodyParser from 'body-parser';
import app from './dev/server';

app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (_req, res) => {
  res.send('The Journal API is running.');
});

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});

export default app;
