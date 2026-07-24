use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{Shutdown, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
};

struct JinaServer {
    child: Child,
    port: u16,
}

impl JinaServer {
    fn start() -> Self {
        let port = TcpListener::bind(("127.0.0.1", 0))
            .expect("ephemeral port")
            .local_addr()
            .expect("local address")
            .port();
        let script =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/jina_mlx_embeddings_server.py");
        let mut child = Command::new("python3")
            .env("CLAWGALLERY_VDR_JINA_MLX_FAKE", "1")
            .arg(script)
            .args(["--port", &port.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("Jina server should start");
        let stdout = child.stdout.take().expect("server stdout");
        let mut ready = String::new();
        BufReader::new(stdout)
            .read_line(&mut ready)
            .expect("server readiness line");
        assert!(ready.contains("\"backend\": \"jina-mlx\""), "got: {ready}");
        Self { child, port }
    }

    fn send(&self, content_type: &str, origin: Option<&str>, body: &str) -> String {
        let mut stream =
            TcpStream::connect(("127.0.0.1", self.port)).expect("server should accept requests");
        let origin_header = origin
            .map(|value| format!("Origin: {value}\r\n"))
            .unwrap_or_default();
        write!(
            stream,
            "POST /embed HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nContent-Type: {}\r\n{}Content-Length: {}\r\n\r\n{}",
            self.port,
            content_type,
            origin_header,
            body.len(),
            body
        )
        .expect("request should write");
        stream.shutdown(Shutdown::Write).expect("request shutdown");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("response should read");
        response
    }
}

impl Drop for JinaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn jina_server_rejects_cross_origin_requests() {
    // Given: the loopback-only Jina embedding server.
    let server = JinaServer::start();

    // When: a browser-originated request targets a local image path.
    let response = server.send(
        "application/json",
        Some("https://attacker.example"),
        r#"{"inputs":[{"kind":"image","role":"query","value":"/etc/passwd"}]}"#,
    );

    // Then: the request is rejected before the path can be processed.
    assert!(response.starts_with("HTTP/1.0 403"), "got: {response}");
    assert!(response.contains("\"error\""), "got: {response}");
}

#[test]
fn jina_server_rejects_non_json_requests() {
    // Given: the managed Jina embedding server.
    let server = JinaServer::start();

    // When: a simple browser content type bypasses JSON preflight.
    let response = server.send(
        "text/plain",
        None,
        r#"{"inputs":[{"kind":"text","role":"query","value":"x"}]}"#,
    );

    // Then: the boundary rejects the media type.
    assert!(response.starts_with("HTTP/1.0 415"), "got: {response}");
    assert!(response.contains("\"error\""), "got: {response}");
}

#[test]
fn jina_server_returns_json_for_malformed_requests() {
    // Given: the managed Jina embedding server.
    let server = JinaServer::start();

    // When: malformed JSON reaches the HTTP boundary.
    let response = server.send("application/json", None, "{");

    // Then: the server returns a stable JSON client error instead of dropping the connection.
    assert!(response.starts_with("HTTP/1.0 400"), "got: {response}");
    assert!(response.contains("\"error\""), "got: {response}");
}
