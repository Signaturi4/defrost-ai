# API

## Invoices

It creates an invoice from tracked hours. Call createInvoice with the client id and the period; it returns the invoice id. The function rounds hours to 15 minutes using ROUNDING_MINUTES which defaults to 15. If the client has no rate set, createInvoice raises MissingRateError and nothing is saved.
