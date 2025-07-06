# Surfside Code Sample

## Introduction

This code comes from a journaling app that I'm currently developing.
The plan is to utilise AWS serverless infrastructure to run the application, as the free
tier is more than enough to support it, even at moderate scale.

## Structure and Content

### Onion Architecture and CQRS

The structure of the project follows the principles of onion architecture, isolating each layer as can be seen in the folder structure. Along with this,
there are some small [CQRS-ish](https://martinfowler.com/bliki/CQRS.html) pieces, which for now share the same data source, but with this structure, allows for easier splitting in
future if needed.

### Domain Driven Design

The code follows DDD principles, with the [aggregate](https://martinfowler.com/bliki/DDD_Aggregate.html) being the Journal itself.

### Use Case

This comes from the Clean Architecture book, a small experiment that I wanted to try out, ultimately it's similar to any other
pattern of using the application layer as an orchestrator, I do like it though. There's some nice things you can do with combining use cases
to build a "feature", i.e. sign up user -> create user use case & create journal. Google search AI does a very nice summary.

> "In Clean Architecture, use cases represent the application-specific business
> rules and orchestrate the flow of data between different layers of the application. They define what the application does and how it interacts with core business logic (entities).
> Use cases are essentially descriptions of intent, outlining the steps needed to achieve a specific goal or functionality."

### Single Table Design in DynamoDB

[Article explaining what this is](https://www.alexdebrie.com/posts/dynamodb-single-table/).<br>
Honestly, I'm not sure I like this style, it's just very different to working with data than a structured DB like Postgres (what I've mainly worked with in my engineering life).
It's an interesting idea, but I want to make it clear that it's an experiment for me.

### Saving based on events

Again, this was just a bit of an experiment to do something along the lines of projections. I've seen it a few times where the aggregate loads all the entities,
one is updated and you save all the entities, rather than just the one that was updated (as you should save the aggregate and that repository is responsible for the
entities). The saving based on uncommitted events on the aggregate seemed like an interesting solution to this.

### Event driven

I've included a use case for analysing the Journal Entry, which utilises an LLM - I've not included the actual LLM implementation, but the stubbed instance that I use
for testing locally. This Lambda is triggered through SQS, which has a filter for this. I'm using the outbox pattern for the event system, it gets called when the `Journal Entry Added`
event occurs.

### Dependency Injection

Using the library [tsyringe](https://github.com/microsoft/tsyringe) for this, I've used a couple of different ones in the past, but I'm a fan of DI.

### Tests

I've included an example test for the CreateJournalEntry command use case, to demonstrate the style of unit testing that I would usually
do, this can be found [here](src/application/useCases/__tests__/command/CreateJournalEntryUseCase.test.ts). To be clear, I'm a strong believer in
all code having a unit test covering it as a minimum (I've just included one of them), with clear isolation in the tests.

## Running the Code

### Requirements

- [Docker](https://www.docker.com/get-started/)

As local serverless development can be a pain, I've created a dev folder, which includes the [server](src/dev/server.ts) and [dynamodb creation](src/dev/createTable.ts), as well as seeding with a
empty test journal. Along with this, there is a JSON dump for a [postman collection](postmanCollection/Journal.postman.json) that allows you to create a journal entry
and query for all journal entries between two given dates.

To run the project, you just need to run 
`docker compose up`, this will set up the server and dynamodb.

If there's any issues with the [postman collection](postmanCollection/Journal.postman.json), the routes are

Get Journal Entries Between Dates `localhost:3000/users/:userId/journals/:journalId/entries?from=2025-07-04&to=2025-07-05`

Create Journal Entry `localhost:3000/users/:userId/journals/:journalId/entries`

and the values for the user and journal are

```
const testUser = '6baa4f27-c2b9-48ce-aeec-adad87835e7e'
const testJournal = '45009cdb-2827-4f8c-9ba7-670cfb05af01'
```
