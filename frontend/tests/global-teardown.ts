export default async function globalTeardown() {
  try {
    await fetch("http://127.0.0.1:8013/api/system/test-shutdown", { method: "POST" });
  } catch {
    // The server may already have stopped after a startup or test failure.
  }
}
