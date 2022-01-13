# TLS

TLS stands for Transport Layer Security. We use TLS to encrypt all traffic
between all services in our kubernetes (k8s) cluster.

All traffic in k8s is intended to be encrypted. [PR
8561](https://github.com/hail-is/hail/pull/8561) started this work. This
document is an adaptation of the explanation given there. Traffic only enters
our cluster via the gateway service. This document describes the use of
encryption within the cluster, but the gateway service also uses TLS to encrypt
communications between it and the outside world.

Currently, all traffic in our cluster should be encrypted except for:
- from the batch-driver to the batch workers
- from the batch workers to the internal-gateway
- to ukbb-rg
- from the notebook service to the notebook workers
- to letsencrypt (oh the irony)

Known issues:
- We have not implemented mutual TLS (mTLS): servers do not verify they trust
  their clients.
- We do not rotate certificates.
- All certificates are signed by one root certificate which everyone trusts. We
  intend for each client and server to have an explicit incoming and outgoing
  trusted clients list. Such lists would prevent a compromised server or client
  from being used to communicate with arbitrary services.
- Our servers should reject all insecure cipher suites and reject TLS versions
  other than 1.3

## HTTPS and TLS

HTTP is implemented on TCP/IP. HTTPS is also implemented on TCP/IP and differs
very mildly. After the socket is opened, a TLS [1] connection is established
over the socket. Thereafter, every HTTP message is encrypted and transported by
the TLS machinery. The HTTP protocol is unchanged. The default port for HTTP is
80 and the default port for HTTPS 443, however any port may be used.

There are currently four versions of TLS, the latest is TLS 1.3. All versions of
SSL are considered insecure.

The OpenSSL library implements TLS. There are other implementations, such as
LibreSSL, but they implement roughly the same interface as OpenSSL.

The TLS protocol defines both a scheme for the *encryption of messages* and for
the *authentication of parties*. The protocol defines authentication as
optional. In practice, at least one party presents authentication. For example,
public web servers authenticate themselves to clients but clients do not
reciprocate.

### Authentication
#### X.509 Certificates

TLS uses X.509 Certificates for authentication. X.509 is a rather complicated
standard. X.509 Certificates can be serialized in a variety of ways. We use the
Privacy-enhanced Electronic Mail (PEM) file format for serialization. PEM is
really simple. A file may contain multiple base64-encoded blobs each with a
header and footer of the form:

```
-----BEGIN LABEL-----
...
-----END LABEL-----
```

where `LABEL` describes the data. We only use two labels: `CERTIFICATE` and
`PRIVATE KEY`.

An X.509 Certificate is an unforgeable proof of identity. It usually is paired
with a private key that was used to digitally sign the certificate. In the
security literature, an authenticatable entity is usually called a
*principal*. Each principal should have a unique private key. In our system the
principals are both our services (e.g. `batch`, `batch-driver`) and any
non-serving clients (e.g. `test-batch`, `admin-pod`). A key and certificate are
generated ad nihilum by `openssl req -new`:

```
openssl req -new \
        -x509
        -keyout key_file
        -out cert_file
        -newkey rsa:4096
        -nodes
        -subj /CN=example.com
        -addext subjectAltName = DNS:www.example.com,DNS:foo.com
```

The first three arguments are self-explanatory. I explain the rest:

- `-newkey rsa:4096`. TLS supports many different kinds of private keys. This
  generates a 4096-bit private RSA key.
- `-nodes`. This should be read as "no DES". It means that the private key is not
  itself encrypted using DES and a password.
- `-subj /CN=example.com`. This certificate is valid for a server whose DNS name
  is `example.com`. If "hostname checking" is enabled (web browsers always
  enable it), then the client will reject the certificate if the hostname used
  to open the socket does not match the certificate.
- `-addext subjectAltName = ...`. This specifies additional acceptable hostnames
  for the aformentioned hostname check.

#### Trust

An X.509 Certificate only proves that the principal has a private key. On the
world wide web, ownership of a particular domain, e.g. `example.com`, is
guaranteed by a Certificate Authority (CA). A Certificate Authority (like
letsencypt or VeriSign) digitally signs the certificate of a user that has
proven they own the domain name mentioned in the certificate (see above
discussion of `CN` and `subjectAltName`). Web browsers have a file of the "root"
certificates of the public Certificate Authorities. They establish a "chain of
trust" from a root certificate to the server's certificate.

Our system is similar. We have a root certificate named
`ssl-config-hail-root`. Each principal has a certificate which is signed by the
root certificate. All other principals trust the root certificate and thus trust
every other principal.

In the future, there will be no root certificate. Instead, each principal will
have a self-signed certificate and a list of trusted certificates. Limiting the
trusted principals is one of many ways in which we limit the damage done by a
misbehaving or compromised service.

### Encryption

TLS has many encryption schemes. I will focus on encryption using a *symmetric*
key because asymmetric schemes do not enable *forward secrecy* [2]. Under
forward secret schemes, the two parties share a private key unknown to all
adversaries. TLS 1.3 makes forward secrecy mandatory. I intend to eventually
require all our services to refuse to speak anything other than TLS 1.3.

The shared private key is used to encrypt and decrypt messages sent over a
socket. This poses a problem: how do two parties who have never met each other
agree on a private key without revealing the key to the public? This is a
classic cryptography problem called [key
exchange](https://en.wikipedia.org/wiki/Key_exchange). The classic solution to
this problem is [Diffie-Hellman key
exchange](https://en.wikipedia.org/wiki/Diffie–Hellman_key_exchange). The
Wikipedia article has "General overview" which is quite clear.

In addition to a key, the parties must agree on a cipher. There are many old,
insecure ciphers available. In the future I intend all our servers to refuse to
use insecure ciphers. Mozilla
[has a list of secure cipher suites](https://wiki.mozilla.org/Security/Server_Side_TLS#Recommended_configurations).

## New Hail Concepts

Every principal in our system has a secret: `ssl-config-NAME`. These secrets are
automatically created for a particular namespace by `tls/create_certs.py`. Who
trusts who (i.e. who is allowed to talk to whom) is defined by
`tls/config.yaml`. For example, `site` is defined in `config.yaml` as follows:

```
- name: site
  domain: site
  kind: nginx
```

A principal named "site" exists. Site's domain names are `site`,
`site.NAMESPACE`, `site.NAMESPACE.svc.cluster.local`. Site's configuration files
should be in NGINX configuration file format. `create_certs.py` will create a
new secret named `ssl-config-site` which contains five files:

- `site-config-http.conf`: an NGINX configuration file that configures TLS for
  incoming requests.
- `site-config-proxy.conf`: an NGINX configuration file that configures TLS for
  outgoing (proxy_pass) requests.
- `site-key.pem`: a private key.
- `site-key-store.p12`: the same private key in PKCS12 format.
- `site-cert.pem`: a certificate.
- `site-incoming.pem`: a list of certificates trusted for incoming requests
  (this currently only contains the root certificate).
- `site-incoming-store.jks`: the same list of trusted incoming certificates in
  Java KeyStore format.
- `site-outgoing.pem`: a list of certificates expected from outgoing requests
  (this currently only contains the root certificate).
- `site-outgoing-store.jks`: the same list of trusted outgoing certificates in
  Java KeyStore format.

If site makes an HTTP request to a server and that server does not return a
certificate in or signed by a certificate in `site-outgoing.pem`, it will
immediately halt the connection.

There are two other kinds: `json` and `curl`. The former is for Hail Python
services. The later is for the admin-pod.

### Batch Confused Deputy

The [confused deputy
problem](https://en.wikipedia.org/wiki/Confused_deputy_problem) is a classic
in computer security. It refers to a situation with two parties: the deputy and
the attacker. The deputy has authority that the attacker does not. For example,
the deputy can arrest people. A *confused deputy* is one which has been tricked
by the attacker into misusing its authority.

Before the TLS PR, Batch had a confused deputy problem: it issues a callback in
response to a batch finishing. That callback is issued from within the cluster
and therefore can name many of our services which are not exposed to the
internet. With the introduction of TLS everywhere, a confused deputy callback
will fail because the victim will not receive a valid Batch certificate (batch
purposely does not send certificates with the callbacks). Batch only uses its
certificate to send a callback for CI. This is safe because we control CI and
ensure it is not compromised.

In the long run, I want to fix batch to use an entirely different network for
callbacks.

### Notes of Annoyance

`aiohttp` silently ignores most invalid TLS requests, this makes debugging a TLS
issue difficult.

`aiohttp`'s `raise_for_status` does not include the HTTP body in the
message. NGINX sometimes returns 400 in response to TLS issues with a proxy. It
includes crucial details in the HTTP body. I usually debug these issues by
sshing to the client pod and using curl to manually reproduce the error.

Readiness and liveness probes cannot use HTTP. Although k8s supports HTTPS, it
does not support so-called "mTLS" or "mutual TLS." This is fancy verbiage for
servers that require clients to send trusted certificates. I will eventually require
this. There is a lot of information in GitHub issues and the Istio web pages
about this, but at the end of the day, kubernetes does not support this. TCP
probes are the best we can do. There [was a
PR](kubernetes/kubernetes#61231) to allow httpGet probes
to send the kubelet certificate, but it was closed because, apparently, the
[httpGet probes can be targeted at arbitrary IP
addresses](kubernetes/kubernetes#61231 (review)) (what
the hell?), ergo Confused Deputy.

### Footnotes

[1] TLS: Transport Layer Security. Preceded by Secure Sockets Layer (SSL) which
    is not obsolete and insecure. After SSL version 3, a new version of SSL was
    proposed in RFC 2246. This new version was backwards-incompatible and was
    thus given a new name: Transport Layer Security. In common discussion, SSL
    and TLS are used interchangeable. Indeed, the python TLS library is called
    `ssl`.

[2] Forward secrecy is a property of an encryption system. Forward secrecy means
    a message cannot be decrypted in the future by an adversary who learned one
    of the private keys. For example, imagine you are sending sensitive messages
    to another individual. If that individual is later coerced into revealing
    their secret key, forward secrecy would prevent the coercer from reading
    your messages. Forward secrecy is achieved by negotiating a shared private
    key between the two parties that is only used for a "session" and then
    discarded. If the session key is securely discarded and neither key can
    recreate it without cooperation from the other key, then *one* leaked key is
    insufficient to reveal the messages.
