#!/bin/bash

pushd chat-ui
npm install
npm run dev -- --port 3000 --host 0.0.0.0
