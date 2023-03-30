import aiohttp
import asyncio
from fpl import FPL, models

global players
global fdr

# Main method that gets data.
async def main():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return await fpl.get_players()       

async def getFDR():
    async with aiohttp.ClientSession() as session:
        fpl = FPL(session)
        return await fpl.FDR()

players = asyncio.run(main())
fdr = asyncio.run(getFDR())

i = 0
count = players.count()
for p in players:
    print(chr(27)+'[2j')
    print("Getting player info " + str(i) + " of " + str(count))
    i = i + 1

