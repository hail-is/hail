const expect = require("chai").expect;
const httpMocks = require("node-mocks-http");
const { getAuthToken } = require("../index");

describe("getAuthToken()", function() {
  it("Should extract auth token from header cookie", function() {
    const access_token = "some_great_token";

    const request = httpMocks.createRequest({
      method: "GET",
      headers: {
        cookie: `access_token=${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(access_token);
  });

  it("Should extract auth token from header cookie when multiple cookies present", function() {
    const access_token = "some_great_token_in_a_list_of_multiple";

    const request = httpMocks.createRequest({
      method: "GET",
      headers: {
        cookie: `access_token=${access_token}; some_other_token=some_great_value`
      }
    });

    expect(getAuthToken(request)).to.be.equal(access_token);
  });

  it("Should extract auth token from header Authorization string", function() {
    const access_token = "some_great_token1";

    const request = httpMocks.createRequest({
      method: "GET",
      headers: {
        Authorization: `Bearer ${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(access_token);
  });

  it("Should treat everything after 'Bearer ' the bearer token value", function() {
    const access_token = `some_great_token1; Basic blah_blah_blah`;

    const request = httpMocks.createRequest({
      method: "GET",
      headers: {
        Authorization: `Bearer ${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(access_token);
  });

  it("Should not extract auth token from header query string", function() {
    const access_token = "some_gre2t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      query: {
        access_token
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not allow 2 methods", function() {
    const access_token = "some_gre3t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      query: {
        access_token
      },
      headers: {
        Authorization: `Bearer ${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not allow 2 methods: query string + authorization header", function() {
    const access_token = "some_gre4t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      query: {
        access_token
      },
      headers: {
        Authorization: `Bearer ${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not allow 2 methods: query string + cookie", function() {
    const access_token = "some_gre5t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      query: {
        access_token
      },
      headers: {
        cookie: `access_token=${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not allow 2 methods: auth header + cookie", function() {
    const access_token = "some_gre6t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      headers: {
        Authorization: `Bearer ${access_token}`,
        cookie: `access_token=${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not allow 3 methods", function() {
    const access_token = "some_gre6t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      query: {
        access_token
      },
      headers: {
        Authorization: `Bearer ${access_token}`,
        cookie: `access_token=${access_token}`
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not read access_token in POST body", function() {
    const access_token = "some_gre6t_token";

    const request = httpMocks.createRequest({
      method: "POST",
      body: {
        access_token
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });

  it("Should not read access_token as GET param", function() {
    const access_token = "some_gre6t_token";

    const request = httpMocks.createRequest({
      method: "GET",
      params: {
        access_token
      }
    });

    expect(getAuthToken(request)).to.be.equal(null);
  });
});
