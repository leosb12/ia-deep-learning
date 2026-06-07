import httpx
import asyncio
import json

async def main():
    r = await httpx.AsyncClient().get('http://localhost:8080/api/politicas', headers={'X-Admin-User-Id': '69e14cac2cfebaf1b3914aa8'})
    data = r.json()
    content = data.get('content', []) if isinstance(data, dict) else data
    print(json.dumps(content[:2], indent=2, ensure_ascii=False))

if __name__ == '__main__':
    asyncio.run(main())
