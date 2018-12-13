let posts = [
  {
    id: '1',
    title: 'Hello World',
    votes: 1,
    url: 'something.com',
    createdAt: new Date().getTime()
  },
  {
    id: '2',
    title: 'Hello World 2',
    votes: 2,
    url: 'something2.com',
    createdAt: new Date().getTime()
  }
];

let messages = [
  {
    id: '1',
    text: 'Hello World',
    userId: 1
  },
  {
    id: '2',
    text: 'By World',
    userId: 2
  }
];

const users = {
  1: {
    id: '1',
    username: 'Robin Wieruch',
    firstname: 'Robin',
    lastname: 'Somethings'
  },
  2: {
    id: '2',
    username: 'Dave Davids',
    firstname: 'Dave',
    lastname: 'Stuffs'
  }
};

// const me = users[1];

const postResolver = {
  Query: {
    me: (parent, args, { me }) => me,
    user: (parent, { id }) => users[id],
    // expects array
    users: () => Object.values(users),
    message: (parent, { id }) => messages[id],
    messages: () => messages,
    allPosts: (parent, { first, skip, orderBy }) => {
      console.info('orderBy', orderBy);
      return posts;
    },
    _allPostsMeta: () => {
      return { count: posts.length };
    }
  },
  User: {
    username: user => `${user.firstname} ${user.lastname}`
  },
  Message: {
    user: (parent, args, context) => {
      return users[parent.userId];
    }
  }
};

module.exports = postResolver;
