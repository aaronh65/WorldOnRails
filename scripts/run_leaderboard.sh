#!/bin/bash

export DEBUG_CHALLENGE=0 # DO NOT MODIFY

# leaderboard and agent config
TEAM_CONFIG=$SAVE_ROOT/config.yml
TEAM_AGENT=$PROJECT_ROOT/autoagents/$AGENT
ROUTE_PATH=$PROJECT_ROOT/assets/routes_$SPLIT/$ROUTE_NAME.xml
SCENARIOS=$PROJECT_ROOT/assets/all_towns_traffic_scenarios.json

CHECKPOINT_ENDPOINT="$SAVE_ROOT/logs/${ROUTE_NAME}.json"

python -m leaderboard.leaderboard_evaluator \
--track=${TRACK} \
--scenarios=${SCENARIOS} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTE_PATH} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${WORLD_PORT} \
--trafficManagerPort=${TM_PORT} \
--debug=${DEBUG_CHALLENGE} \
--repetitions=${REPETITIONS}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

